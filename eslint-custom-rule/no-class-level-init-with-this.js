export default {
  meta: {
    type: "problem",
    docs: {
      description:
        "Disallow `this` usage in class field initializers for dependent instance creation",
      recommended: false,
    },
    messages: {
      noThisReference:
        "Do not reference `this` in class field initializers when creating a new instance. Move it to the constructor instead.",
    },
    schema: [],
  },

  create(context) {
    function containsThisExpression(node) {
      if (!node || typeof node !== "object") return false;
      if (
        node.type === "ArrowFunctionExpression" ||
        node.type === "FunctionExpression"
      ) {
        return false;
      }
      if (node.type === "ThisExpression") return true;

      if (node.type === "CallExpression") {
        // If we have a call, we check the presence of "this" in the arguments
        for (const argument of node.arguments) {
          if (containsThisExpression(argument)) return true;
        }
        // We check if the callee is not a lambda that is directly called
        const callee = node.callee;
        if (
          callee.type === "ArrowFunctionExpression" &&
          containsThisExpression(callee.body)
        ) {
          return true;
        }
        // We visited the arguments and the callee, no need to go further
        return false;
      }

      // We walk down the AST for other nodes
      for (const key of Object.keys(node)) {
        if (key === "parent") {
          return false;
        }
        const value = node[key];

        if (Array.isArray(value)) {
          for (const el of value) {
            if (containsThisExpression(el)) return true;
          }
        } else if (value && typeof value === "object") {
          if (containsThisExpression(value)) return true;
        }
      }

      return false;
    }

    return {
      // For class fields (in modern JS/TS)
      PropertyDefinition(node) {
        if (!node.value) return;
        if (containsThisExpression(node.value)) {
          context.report({
            node,
            messageId: "noThisReference",
          });
        }
      },
    };
  },
};
